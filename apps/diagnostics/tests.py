from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from .serializers import SequenceCreateSerializer
from .services.inference import InferenceService
from .services.preprocess import (
	is_supported_sequence_filename,
	extract_sequences_from_text,
)


class PreprocessTests(SimpleTestCase):
	def test_extract_sequences_from_text_splits_fasta_records(self):
		raw_text = (
			'>seq1\n'
			'AATCCAGCATTAAGAGAAGAGATGCGTATTCAATTCAATGACATGAACAGTGCCCTTACA\n'
			'>seq2\n'
			'ATCGATGAAAATATCTCTGCTTTTTCTTTTTTTATTTGGTATATGCTTTACTTGTAATCG\n'
			'AAAATAAAGCACTAATAAGAGTATTTATAGGTGTTTGAAGTTATTTCAGT\n'
			'>seq3\n'
			'TATTAGTGCTTTATTTTCGATTACAAGTAAAGCATATACCAAATAAAAAAAGAAAAAGCA\n'
			'GTCGTATAA\n'
		)

		sequences = extract_sequences_from_text(raw_text)

		self.assertEqual(len(sequences), 3)
		self.assertEqual(sequences[0], 'AATCCAGCATTAAGAGAAGAGATGCGTATTCAATTCAATGACATGAACAGTGCCCTTACA')
		self.assertTrue(sequences[1].startswith('ATCGATGAAAATATCTCTGCTTTTTCTTTTTTTATTTGGTATATGCTTTACTTGTAATCG'))
		self.assertEqual(sequences[2], 'TATTAGTGCTTTATTTTCGATTACAAGTAAAGCATATACCAAATAAAAAAAGAAAAAGCAGTCGTATAA')

	def test_extract_sequences_from_text_handles_glued_headers(self):
		raw_text = 'ATCGATGAAAATATCTCTGCTTTTTCTTTTTTTATTTGGTATATGCTTTACTTGTAATCG>seq2\nTTCTAATAAAGGGAATCTCATCAAAAATCCATATAGTAACCCCAACGAAAAATTAATCAA'

		sequences = extract_sequences_from_text(raw_text)

		self.assertEqual(len(sequences), 2)
		self.assertEqual(sequences[0], 'ATCGATGAAAATATCTCTGCTTTTTCTTTTTTTATTTGGTATATGCTTTACTTGTAATCG')
		self.assertEqual(sequences[1], 'TTCTAATAAAGGGAATCTCATCAAAAATCCATATAGTAACCCCAACGAAAAATTAATCAA')

	def test_supported_sequence_file_extensions(self):
		self.assertTrue(is_supported_sequence_filename('sample.FASTA'))
		self.assertTrue(is_supported_sequence_filename('sample.fna'))
		self.assertTrue(is_supported_sequence_filename('sample.fa'))
		self.assertTrue(is_supported_sequence_filename('sample.txt'))
		self.assertFalse(is_supported_sequence_filename('sample.csv'))

	def test_file_upload_serializer_accepts_supported_extensions(self):
		for filename in ('sample.fasta', 'sample.fna', 'sample.txt'):
			with self.subTest(filename=filename):
				serializer = SequenceCreateSerializer(data={
					'seq_nom': 'sample',
					'seq_nom_courant': 'sample',
					'file': SimpleUploadedFile(filename, b'>sequence\nACGTN\n'),
				})

				self.assertTrue(serializer.is_valid(), serializer.errors)


class InferenceAttentionTests(SimpleTestCase):
	def setUp(self):
		self.service = InferenceService.__new__(InferenceService)

	def test_no_attention_is_returned_when_gmo_rate_is_zero(self):
		attention = self.service._build_attention_segments('ATATATATCCCCGGGG', 0.0)

		self.assertEqual(attention, [])

	def test_attention_is_returned_when_gmo_rate_is_positive(self):
		attention = self.service._build_attention_segments('ATATATATCCCCGGGG', 0.01)

		self.assertTrue(attention)

	def test_all_attention_segments_are_returned(self):
		sequence = 'AT' * 100
		attention = self.service._build_attention_segments(sequence, 0.01)

		self.assertEqual(len(attention), 21)

	def test_occlusion_ranks_windows_by_probability_drop(self):
		def predictor(sequence):
			probability = 0.8 if 'G' in sequence[:32] else 0.2
			return {'probabilities': {'gmo': probability}}

		attention = self.service._build_occlusion_segments(
			'G' * 32 + 'A' * 32,
			0.8,
			predictor,
		)

		self.assertEqual(attention[0]['start'], 0)
		self.assertGreater(attention[0]['impact'], 0)
		self.assertEqual(attention[0]['score'], abs(attention[0]['impact']))


class InferenceChunkingTests(SimpleTestCase):
	def setUp(self):
		self.service = InferenceService.__new__(InferenceService)

	def test_long_sequence_predictions_are_combined(self):
		def predictor(chunk):
			return {
				'label': 'gmo',
				'score': 0.8,
				'probabilities': {'gmo': 0.8, 'no_gmo': 0.2},
				'attention': [{'start': 0, 'end': 4, 'sequence': chunk[:4], 'score': 0.8}],
				'in_uncertainty_zone': False,
			}

		sequence = 'A' * 2050
		result = self.service._predict_in_chunks(sequence, predictor)

		self.assertEqual(result['chunks_processed'], 2)
		self.assertEqual(result['probabilities']['gmo'], 0.8)
		self.assertEqual([item['start'] for item in result['attention']], [0, 2048])

	def test_combined_zero_gmo_probability_has_no_attention(self):
		def predictor(chunk):
			return {
				'label': 'no-gmo',
				'score': 0.9999,
				'probabilities': {'gmo': 0.00001, 'no_gmo': 0.99999},
				'attention': [{'start': 0, 'end': 4, 'sequence': chunk[:4], 'score': 1.0}],
				'in_uncertainty_zone': False,
			}

		result = self.service._predict_in_chunks('A' * 2050, predictor)

		self.assertEqual(result['probabilities']['gmo'], 0.0)
		self.assertEqual(result['attention'], [])
